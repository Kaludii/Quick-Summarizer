from transformers import pipeline
import gradio as gr
from gradio.mix import Parallel, Series

quick_sum = gr.Interface.load('huggingface/Kaludi/Quick-Summarization')                
        
example1 = """British Prime Minister Theresa May said on Friday she would continue to govern in the interests of all Northern Ireland and uphold the agreement that ended decades of sectarian violence in the province. The statement comes as an impasse over the future of the Irish border once Britain leaves the European Union looked to have been resolved. This Government will continue to govern in the interests of the whole community in Northern Ireland and uphold the Agreements that have underpinned the huge progress that has been made over the past two decades, a statement published on the government s website said.
"""

example2 = '''What is a paragraph?
Paragraphs are the building blocks of papers. Many students define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. A paragraph is defined as “a group of sentences or a single sentence that forms a unit” (Lunsford and Connors 116). Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long. Ultimately, a paragraph is a sentence or group of sentences that support one main idea. In this handout, we will refer to this as the “controlling idea,” because it controls what happens in the rest of the paragraph.'''


samples = [[example1],[example2]]

iface = Parallel(quick_sum,
                 theme='huggingface', 
                 title= 'Quick Summarizer App', 
                 description = "This is a Text Summarization Model that has been trained by <strong><a href='https://huggingface.co/Kaludi'>Kaludi</a></strong> to Transform long and complex texts into concise and meaningful summaries. Get a quick and accurate overview of any document in seconds, saving you time and effort.",
                 article = "<p style='text-align: center'><a href='https://github.com/Kaludii'>Github</a> | <a href='https://huggingface.co/Kaludi'>HuggingFace</a></p>",
                 examples=samples,
                 inputs = gr.inputs.Textbox(lines = 8, label="Text"))

iface.launch(inline = False)