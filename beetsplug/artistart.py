# This file is part of beets.
# Copyright 2016, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""Fetches album art.
"""

import os
import re
from collections import OrderedDict
from contextlib import closing
from tempfile import NamedTemporaryFile

import confuse
import requests
from mediafile import image_mime_type
import shutil

from beets import config, importer, plugins, ui, util
from beets.util import bytestring_path, py3_path, sorted_walk, syspath
from beets.util.artresizer import ArtResizer

try:
    from bs4 import BeautifulSoup

    HAS_BEAUTIFUL_SOUP = True
except ImportError:
    HAS_BEAUTIFUL_SOUP = False


CONTENT_TYPES = {"image/jpeg": [b"jpg", b"jpeg"], "image/png": [b"png"]}
IMAGE_EXTENSIONS = [ext for exts in CONTENT_TYPES.values() for ext in exts]


class Candidate:
    """Holds information about a matching artwork, deals with validation of
    dimension restrictions and resizing.
    """

    CANDIDATE_BAD = 0
    CANDIDATE_EXACT = 1
    CANDIDATE_DOWNSCALE = 2
    CANDIDATE_DOWNSIZE = 3
    CANDIDATE_DEINTERLACE = 4
    CANDIDATE_REFORMAT = 5

    MATCH_EXACT = 0
    MATCH_FALLBACK = 1

    def __init__(
        self, log, path=None, url=None, source="", match=None, size=None
    ):
        self._log = log
        self.path = path
        self.url = url
        self.source = source
        self.check = None
        self.match = match
        self.size = size

    def _validate(self, plugin):
        """Determine whether the candidate artwork is valid based on
        its dimensions (width and ratio).

        Return `CANDIDATE_BAD` if the file is unusable.
        Return `CANDIDATE_EXACT` if the file is usable as-is.
        Return `CANDIDATE_DOWNSCALE` if the file must be rescaled.
        Return `CANDIDATE_DOWNSIZE` if the file must be resized, and possibly
            also rescaled.
        Return `CANDIDATE_DEINTERLACE` if the file must be deinterlaced.
        Return `CANDIDATE_REFORMAT` if the file has to be converted.
        """
        if not self.path:
            return self.CANDIDATE_BAD

        if not (
            plugin.enforce_ratio
            or plugin.minwidth
            or plugin.maxwidth
            or plugin.max_filesize
            or plugin.deinterlace
            or plugin.cover_format
        ):
            return self.CANDIDATE_EXACT

        # get_size returns None if no local imaging backend is available
        if not self.size:
            self.size = ArtResizer.shared.get_size(self.path)
        self._log.debug("image size: {}", self.size)

        if not self.size:
            self._log.warning(
                "Could not get size of image (please see "
                "documentation for dependencies). "
                "The configuration options `minwidth`, "
                "`enforce_ratio` and `max_filesize` "
                "may be violated."
            )
            return self.CANDIDATE_EXACT

        short_edge = min(self.size)
        long_edge = max(self.size)

        # Check minimum dimension.
        if plugin.minwidth and self.size[0] < plugin.minwidth:
            self._log.debug(
                "image too small ({} < {})", self.size[0], plugin.minwidth
            )
            return self.CANDIDATE_BAD

        # Check aspect ratio.
        edge_diff = long_edge - short_edge
        if plugin.enforce_ratio:
            if plugin.margin_px:
                if edge_diff > plugin.margin_px:
                    self._log.debug(
                        "image is not close enough to being "
                        "square, ({} - {} > {})",
                        long_edge,
                        short_edge,
                        plugin.margin_px,
                    )
                    return self.CANDIDATE_BAD
            elif plugin.margin_percent:
                margin_px = plugin.margin_percent * long_edge
                if edge_diff > margin_px:
                    self._log.debug(
                        "image is not close enough to being "
                        "square, ({} - {} > {})",
                        long_edge,
                        short_edge,
                        margin_px,
                    )
                    return self.CANDIDATE_BAD
            elif edge_diff:
                # also reached for margin_px == 0 and margin_percent == 0.0
                self._log.debug(
                    "image is not square ({} != {})", self.size[0], self.size[1]
                )
                return self.CANDIDATE_BAD

        # Check maximum dimension.
        downscale = False
        if plugin.maxwidth and self.size[0] > plugin.maxwidth:
            self._log.debug(
                "image needs rescaling ({} > {})", self.size[0], plugin.maxwidth
            )
            downscale = True

        # Check filesize.
        downsize = False
        if plugin.max_filesize:
            filesize = os.stat(syspath(self.path)).st_size
            if filesize > plugin.max_filesize:
                self._log.debug(
                    "image needs resizing ({}B > {}B)",
                    filesize,
                    plugin.max_filesize,
                )
                downsize = True

        # Check image format
        reformat = False
        if plugin.cover_format:
            fmt = ArtResizer.shared.get_format(self.path)
            reformat = fmt != plugin.cover_format
            if reformat:
                self._log.debug(
                    "image needs reformatting: {} -> {}",
                    fmt,
                    plugin.cover_format,
                )

        if downscale:
            return self.CANDIDATE_DOWNSCALE
        elif downsize:
            return self.CANDIDATE_DOWNSIZE
        elif plugin.deinterlace:
            return self.CANDIDATE_DEINTERLACE
        elif reformat:
            return self.CANDIDATE_REFORMAT
        else:
            return self.CANDIDATE_EXACT

    def validate(self, plugin):
        self.check = self._validate(plugin)
        return self.check

    def resize(self, plugin):
        if self.check == self.CANDIDATE_DOWNSCALE:
            self.path = ArtResizer.shared.resize(
                plugin.maxwidth,
                self.path,
                quality=plugin.quality,
                max_filesize=plugin.max_filesize,
            )
        elif self.check == self.CANDIDATE_DOWNSIZE:
            # dimensions are correct, so maxwidth is set to maximum dimension
            self.path = ArtResizer.shared.resize(
                max(self.size),
                self.path,
                quality=plugin.quality,
                max_filesize=plugin.max_filesize,
            )
        elif self.check == self.CANDIDATE_DEINTERLACE:
            self.path = ArtResizer.shared.deinterlace(self.path)
        elif self.check == self.CANDIDATE_REFORMAT:
            self.path = ArtResizer.shared.reformat(
                self.path,
                plugin.cover_format,
                deinterlaced=plugin.deinterlace,
            )


def _logged_get(log, *args, **kwargs):
    """Like `requests.get`, but logs the effective URL to the specified
    `log` at the `DEBUG` level.

    Use the optional `message` parameter to specify what to log before
    the URL. By default, the string is "getting URL".

    Also sets the User-Agent header to indicate beets.
    """
    # Use some arguments with the `send` call but most with the
    # `Request` construction. This is a cheap, magic-filled way to
    # emulate `requests.get` or, more pertinently,
    # `requests.Session.request`.
    req_kwargs = kwargs
    send_kwargs = {}
    for arg in ("stream", "verify", "proxies", "cert", "timeout"):
        if arg in kwargs:
            send_kwargs[arg] = req_kwargs.pop(arg)

    # Our special logging message parameter.
    if "message" in kwargs:
        message = kwargs.pop("message")
    else:
        message = "getting URL"

    req = requests.Request("GET", *args, **req_kwargs)

    with requests.Session() as s:
        s.headers = {"User-Agent": "beets"}
        prepped = s.prepare_request(req)
        settings = s.merge_environment_settings(
            prepped.url, {}, None, None, None
        )
        send_kwargs.update(settings)
        log.debug("{}: {}", message, prepped.url)
        return s.send(prepped, **send_kwargs)


class RequestMixin:
    """Adds a Requests wrapper to the class that uses the logger, which
    must be named `self._log`.
    """

    def request(self, *args, **kwargs):
        """Like `requests.get`, but uses the logger `self._log`.

        See also `_logged_get`.
        """
        return _logged_get(self._log, *args, **kwargs)


# ART SOURCES ################################################################


class ArtSource(RequestMixin):
    VALID_MATCHING_CRITERIA = ["default"]

    def __init__(self, log, config, match_by=None):
        self._log = log
        self._config = config
        self.match_by = match_by or self.VALID_MATCHING_CRITERIA

    @staticmethod
    def add_default_config(config):
        pass

    @classmethod
    def available(cls, log, config):
        """Return whether or not all dependencies are met and the art source is
        in fact usable.
        """
        return True

    def get_artist(self, album, plugin, paths):
        raise NotImplementedError()

    def _candidate(self, **kwargs):
        return Candidate(source=self, log=self._log, **kwargs)

    def fetch_image(self, candidate, plugin):
        raise NotImplementedError()

    def cleanup(self, candidate):
        pass


class LocalArtSource(ArtSource):
    IS_LOCAL = True
    LOC_STR = "local"

    def fetch_image(self, candidate, plugin):
        pass


class RemoteArtSource(ArtSource):
    IS_LOCAL = False
    LOC_STR = "remote"

    def fetch_image(self, candidate, plugin):
        """Downloads an image from a URL and checks whether it seems to
        actually be an image. If so, returns a path to the downloaded image.
        Otherwise, returns None.
        """
        if plugin.maxwidth:
            candidate.url = ArtResizer.shared.proxy_url(
                plugin.maxwidth, candidate.url
            )
        try:
            with closing(
                self.request(
                    candidate.url, stream=True, message="downloading image"
                )
            ) as resp:
                ct = resp.headers.get("Content-Type", None)

                # Download the image to a temporary file. As some servers
                # (notably fanart.tv) have proven to return wrong Content-Types
                # when images were uploaded with a bad file extension, do not
                # rely on it. Instead validate the type using the file magic
                # and only then determine the extension.
                data = resp.iter_content(chunk_size=1024)
                header = b""
                for chunk in data:
                    header += chunk
                    if len(header) >= 32:
                        # The imghdr module will only read 32 bytes, and our
                        # own additions in mediafile even less.
                        break
                else:
                    # server didn't return enough data, i.e. corrupt image
                    return

                real_ct = image_mime_type(header)
                if real_ct is None:
                    # detection by file magic failed, fall back to the
                    # server-supplied Content-Type
                    # Is our type detection failsafe enough to drop this?
                    real_ct = ct

                if real_ct not in CONTENT_TYPES:
                    self._log.debug(
                        "not a supported image: {}",
                        real_ct or "unknown content type",
                    )
                    return

                ext = b"." + CONTENT_TYPES[real_ct][0]

                if real_ct != ct:
                    self._log.warning(
                        "Server specified {}, but returned a "
                        "{} image. Correcting the extension "
                        "to {}",
                        ct,
                        real_ct,
                        ext,
                    )

                suffix = py3_path(ext)
                with NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                    # write the first already loaded part of the image
                    fh.write(header)
                    # download the remaining part of the image
                    for chunk in data:
                        fh.write(chunk)
                self._log.debug(
                    "downloaded art to: {0}", util.displayable_path(fh.name)
                )
                candidate.path = util.bytestring_path(fh.name)
                return

        except (OSError, requests.RequestException, TypeError) as exc:
            # Handling TypeError works around a urllib3 bug:
            # https://github.com/shazow/urllib3/issues/556
            self._log.debug("error fetching art: {}", exc)
            return

    def cleanup(self, candidate):
        if candidate.path:
            try:
                util.remove(path=candidate.path)
            except util.FilesystemError as exc:
                self._log.debug("error cleaning up tmp art: {}", exc)


class FanartTV(RemoteArtSource):
    """Art from fanart.tv requested using their API"""

    NAME = "fanart.tv"
    API_URL = "https://webservice.fanart.tv/v3/"
    API_ARTISTS = API_URL + "music/"
    PROJECT_KEY = "61a7d0ab4e67162b7a0c7c35915cd48e"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_key = self._config["fanarttv_key"].get()

    @staticmethod
    def add_default_config(config):
        config.add(
            {
                "fanarttv_key": None,
            }
        )
        config["fanarttv_key"].redact = True

    def get_artist(self, album, plugin, paths):
        if not album.mb_albumartistid:
            return

        try:
            response = self.request(
                self.API_ARTISTS + album.mb_albumartistid,
                headers={
                    "api-key": self.PROJECT_KEY,
                    "client-key": self.client_key,
                },
            )
        except requests.RequestException:
            self._log.debug("fanart.tv: error receiving response")
            return

        try:
            data = response.json()
        except ValueError:
            self._log.debug(
                "fanart.tv: error loading response: {}", response.text
            )
            return

        if "status" in data and data["status"] == "error":
            if "not found" in data["error message"].lower():
                self._log.debug("fanart.tv: no image found")
            elif "api key" in data["error message"].lower():
                self._log.warning(
                    "fanart.tv: Invalid API key given, please "
                    "enter a valid one in your config file."
                )
            else:
                self._log.debug(
                    "fanart.tv: error on request: {}", data["error message"]
                )
            return

        matches = data.get("artistbackground", [])

        matches.sort(key=lambda x: int(x["likes"]), reverse=True)
        for item in matches:
            # fanart.tv has a strict size requirement for album art to be
            # uploaded
            yield self._candidate(
                url=item["url"], match=Candidate.MATCH_EXACT, size=(1920, 1080)
            )


class FileSystem(LocalArtSource):
    NAME = "Filesystem"

    @staticmethod
    def filename_priority(filename, cover_names):
        """Sort order for image names.

        Return indexes of cover names found in the image filename. This
        means that images with lower-numbered and more keywords will have
        higher priority.
        """
        return [idx for (idx, x) in enumerate(cover_names) if x in filename]

    def get_artist(self, album, plugin, paths):
        """Look for album art files in the specified directories."""
        if not paths:
            return
        cover_names = list(map(util.bytestring_path, plugin.cover_names))
        cover_names_str = b"|".join(cover_names)
        cover_pat = rb"".join([rb"(\b|_)(", cover_names_str, rb")(\b|_)"])

        for path in paths:
            if not os.path.isdir(syspath(path)):
                continue

            # Find all files that look like images in the directory.
            images = []
            ignore = config["ignore"].as_str_seq()
            ignore_hidden = config["ignore_hidden"].get(bool)
            for _, _, files in sorted_walk(
                path, ignore=ignore, ignore_hidden=ignore_hidden
            ):
                for fn in files:
                    fn = bytestring_path(fn)
                    for ext in IMAGE_EXTENSIONS:
                        if fn.lower().endswith(b"." + ext) and os.path.isfile(
                            syspath(os.path.join(path, fn))
                        ):
                            images.append(fn)

            # Look for "preferred" filenames.
            images = sorted(
                images, key=lambda x: self.filename_priority(x, cover_names)
            )
            remaining = []
            for fn in images:
                if re.search(cover_pat, os.path.splitext(fn)[0], re.I):
                    self._log.debug(
                        "using well-named art file {0}",
                        util.displayable_path(fn),
                    )
                    yield self._candidate(
                        path=os.path.join(path, fn), match=Candidate.MATCH_EXACT
                    )
                else:
                    remaining.append(fn)

            # Fall back to any image in the folder.
            if remaining and not plugin.cautious:
                self._log.debug(
                    "using fallback art file {0}",
                    util.displayable_path(remaining[0]),
                )
                yield self._candidate(
                    path=os.path.join(path, remaining[0]),
                    match=Candidate.MATCH_FALLBACK,
                )


# Try each source in turn.

# Note that SOURCES_ALL is redundant (and presently unused). However, we keep
# it around nn order not break plugins that "register" (a.k.a. monkey-patch)
# their own fetchart sources.
SOURCES_ALL = [
    "filesystem",
    "fanarttv",
]

ART_SOURCES = {
    "filesystem": FileSystem,
    "fanarttv": FanartTV,
}
SOURCE_NAMES = {v: k for k, v in ART_SOURCES.items()}

# PLUGIN LOGIC ###############################################################


class ArtistArtPlugin(plugins.BeetsPlugin, RequestMixin):
    PAT_PX = r"(0|[1-9][0-9]*)px"
    PAT_PERCENT = r"(100(\.00?)?|[1-9]?[0-9](\.[0-9]{1,2})?)%"

    def __init__(self):
        super().__init__()

        # Holds candidates corresponding to downloaded images between
        # fetching them and placing them in the filesystem.
        self.art_candidates = {}

        self.config.add(
            {
                "auto": True,
                "minwidth": 0,
                "maxwidth": 0,
                "quality": 0,
                "max_filesize": 0,
                "enforce_ratio": False,
                "cautious": False,
                "cover_names": ["cover", "front", "art", "album", "folder"],
                "sources": [
                    "filesystem",
                    "coverart",
                    "itunes",
                    "amazon",
                    "albumart",
                    "cover_art_url",
                ],
                "store_source": False,
                "high_resolution": False,
                "deinterlace": False,
                "cover_format": None,
            }
        )
        for source in ART_SOURCES.values():
            source.add_default_config(self.config)

        self.minwidth = self.config["minwidth"].get(int)
        self.maxwidth = self.config["maxwidth"].get(int)
        self.max_filesize = self.config["max_filesize"].get(int)
        self.quality = self.config["quality"].get(int)

        # allow both pixel and percentage-based margin specifications
        self.enforce_ratio = self.config["enforce_ratio"].get(
            confuse.OneOf(
                [
                    bool,
                    confuse.String(pattern=self.PAT_PX),
                    confuse.String(pattern=self.PAT_PERCENT),
                ]
            )
        )
        self.margin_px = None
        self.margin_percent = None
        self.deinterlace = self.config["deinterlace"].get(bool)
        if type(self.enforce_ratio) is str:
            if self.enforce_ratio[-1] == "%":
                self.margin_percent = float(self.enforce_ratio[:-1]) / 100
            elif self.enforce_ratio[-2:] == "px":
                self.margin_px = int(self.enforce_ratio[:-2])
            else:
                # shouldn't happen
                raise confuse.ConfigValueError()
            self.enforce_ratio = True

        cover_names = self.config["cover_names"].as_str_seq()
        self.cover_names = list(map(util.bytestring_path, cover_names))
        self.cautious = self.config["cautious"].get(bool)
        self.store_source = self.config["store_source"].get(bool)

        self.src_removed = config["import"]["delete"].get(bool) or config[
            "import"
        ]["move"].get(bool)

        self.cover_format = self.config["cover_format"].get(
            confuse.Optional(str)
        )

        if self.config["auto"]:
            # Enable two import hooks when fetching is enabled.
            self.import_stages = [self.fetch_artist_art]
            self.register_listener("import_task_files", self.assign_artist_art)

        available_sources = [
            (s_name, c)
            for (s_name, s_cls) in ART_SOURCES.items()
            if s_cls.available(self._log, self.config)
            for c in s_cls.VALID_MATCHING_CRITERIA
        ]
        sources = plugins.sanitize_pairs(
            self.config["sources"].as_pairs(default_value="*"),
            available_sources,
        )

        if "remote_priority" in self.config:
            self._log.warning(
                "The `fetch_art.remote_priority` configuration option has "
                "been deprecated. Instead, place `filesystem` at the end of "
                "your `sources` list."
            )
            if self.config["remote_priority"].get(bool):
                fs = []
                others = []
                for s, c in sources:
                    if s == "filesystem":
                        fs.append((s, c))
                    else:
                        others.append((s, c))
                sources = others + fs

        self.sources = [
            ART_SOURCES[s](self._log, self.config, match_by=[c])
            for s, c in sources
        ]

    # Asynchronous; after music is added to the library.
    def fetch_artist_art(self, session, task):
        """Find art for the album being imported."""
        if task.is_album:  # Only fetch art for full albums.

            artpath = self.get_art_path(task.album)

            if artpath and os.path.isfile(
                syspath(artpath)
            ):
                # Album already has art (probably a re-import); skip it.
                return
            if task.choice_flag == importer.action.ASIS:
                # For as-is imports, don't search Web sources for art.
                local = True
            elif task.choice_flag in (
                importer.action.APPLY,
                importer.action.RETAG,
            ):
                # Search everywhere for art.
                local = False
            else:
                # For any other choices (e.g., TRACKS), do nothing.
                return

            candidate = self.art_for_artist(task.album, task.paths, local)

            if candidate:
                self.art_candidates[task] = candidate

    def _set_artist_art(self, album, artpath, candidate, delete=False):

        _, candidate_extension = os.path.splitext(candidate.path)
        artfilename, art_extension = os.path.splitext(artpath)
        shutil.copyfile(candidate.path, artfilename + candidate_extension.decode("utf-8") )


    # Synchronous; after music files are put in place.
    def assign_artist_art(self, session, task):
        """Place the discovered art in the filesystem."""
        if task in self.art_candidates:
            candidate = self.art_candidates.pop(task)

            artpath = self.get_art_path(task.album)
            self._set_artist_art(task.album, artpath, candidate, not self.src_removed)

            if self.src_removed:
                task.prune(candidate.path)

    # Manual artwork fetching.
    def commands(self):
        cmd = ui.Subcommand("artistart", help="download artist artwork")
        cmd.parser.add_option(
            "-f",
            "--force",
            dest="force",
            action="store_true",
            default=False,
            help="re-download art when already present",
        )
        cmd.parser.add_option(
            "-q",
            "--quiet",
            dest="quiet",
            action="store_true",
            default=False,
            help="quiet mode: do not output artists that already have artwork",
        )

        def func(lib, opts, args):
            self.batch_fetch_artist_art(
                lib, lib.albums(ui.decargs(args)), opts.force, opts.quiet
            )

        cmd.func = func
        return [cmd]

    # Utilities converted from functions to methods on logging overhaul

    def art_for_artist(self, album, paths, local_only=False):
        """Given an Album object, returns a path to downloaded art for the
        album (or None if no art is found). If `maxwidth`, then images are
        resized to this maximum pixel size. If `quality` then resized images
        are saved at the specified quality level. If `local_only`, then only
        local image files from the filesystem are returned; no network
        requests are made.
        """
        out = None

        for source in self.sources:
            if source.IS_LOCAL or not local_only:
                self._log.debug(
                    "trying source {0} for album {1.albumartist} - {1.album}",
                    SOURCE_NAMES[type(source)],
                    album,
                )
                # URLs might be invalid at this point, or the image may not
                # fulfill the requirements
                for candidate in source.get_artist(album, self, paths):
                    source.fetch_image(candidate, self)
                    if candidate.validate(self):
                        out = candidate
                        self._log.debug(
                            "using {0.LOC_STR} image {1}".format(
                                source, util.displayable_path(out.path)
                            )
                        )
                        break
                    # Remove temporary files for invalid candidates.
                    source.cleanup(candidate)
                if out:
                    break

        if out:
            out.resize(self)

        return out

    def batch_fetch_artist_art(self, lib, albums, force, quiet):
        """Fetch artist art for each of the albums. This implements the manual
        artistart CLI command.
        """
        for album in albums:

            artpath = self.get_art_path(album)
            if (
                artpath
                and not force
                and os.path.isfile(syspath(artpath))
            ):
                if not quiet:
                    message = ui.colorize(
                        "text_highlight_minor", "has artist art"
                    )
                    self._log.info("{0}: {1}", album, message)
            else:
                # In ordinary invocations, look for images on the
                # filesystem. When forcing, however, always go to the Web
                # sources.
                local_paths = None if force else [album.path]

                candidate = self.art_for_artist(album, local_paths)
                if candidate:
                    self._set_artist_art(album, artpath, candidate)
                    message = ui.colorize("text_success", "found artist art")
                else:
                    message = ui.colorize("text_error", "no art found")
                self._log.info("{0}: {1}", album, message)

    def get_art_path(self, album):
        filename = self.config["filename"].as_str()
        return os.path.join(album.path.decode("utf-8"), filename)