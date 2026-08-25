# superstructure
Pipeline to get structured data from Supernote device note files. 

Supernote devices are a pleasure to work on, easy on the eyes, low-powered, resilient devices. I'm betting the last longer in the field than a glass brick like an iPad. This repository contains an archaeological single-context recording sheet (sourced from BAJR) as a demo. The idea is that the underlying template, like an archaeological single-context recording sheet, conveys semantic information by the positioning and layout of its elements. You'd make notes on the supernote device, copy those over to your computer, run the scripts in this repo, and voilà, structured data.

1. turn your pdf into a png file; upload it to `my_styles` on your supernote device. The `make_template.py` script will format for nomad or manta.
2. Make a new note. If you make it a real-time recognition note, the `rtr-parse.py` script is what you'll want later on. Supernote's on-device ocr recognizes the sequence of pen strokes, apparently, and this leads to better quality than say an image-based ocr.
3. Set the png file as a template for the note; add data, pages as desired. 
4. Move the note to your computer.
5. Make sure the parser script knows the area of the bounding boxes by using the `visualize-rois.py` script. This will let you drag and drop the areas visually on the template; it will return the coordinates properly scaled for the device you use (manta or nomad). Copy that data into the parse scripts.
6. `parse_context_record.py` lets you use ocrmac, rapidocr, or a vlm.
7. `parse_hybrid_record.py` retrieves the on-device OCR and the bounding boxes, then uses a VLM to 'correct' errors. **This one currently gives the best results**.

**A work in progress**. Developed on a mac m1. `parse_hybrid_record.py` uses mlx, so you'd have to faff about if you wanted a windows/linux solution; if you're on windows, the `parse_context_record.py` script should probably work for you. But the best idea would be to modify the strategy used by parse_hybrid_record.py to work on windows.

You could always plumb those scripts into a high-end vlm if you want; that should probably give you best results. Anyway, this general approach should work for any kind of form filling you might want to do on a Supernote. 
