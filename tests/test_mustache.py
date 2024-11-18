import importlib.resources

import faery

with importlib.resources.open_text(faery, "cli/faery_script.mustache") as template_file:
    template = template_file.read()

contents = faery.mustache.render(template=template, jobs=[])
