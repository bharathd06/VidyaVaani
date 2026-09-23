from graphviz import Digraph

def generate_diagram():
    dot = Digraph(comment='VidyaVaani Architecture', format='png')
    dot.attr(rankdir='LR', size='10,6')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    dot.node('A', 'Audio Input\n(Microphone / Video)')
    dot.node('B', 'Parallel Queue')
    
    with dot.subgraph(name='cluster_workers') as c:
        c.attr(style='dashed', color='black')
        c.node('C', 'ASR Engine\n(Whisper int8)')
        c.node('O', 'LLM Agent\n(Ollama Domain Refinement)')
        c.node('D', 'NMT Engine\n(IndicTrans2)')
        c.node('E', 'TTS Engine\n(Edge TTS)')
        c.attr(label='Background Worker Thread')
        
    dot.node('F', 'Audio Output\n(Speaker / Dubbed Video)')

    dot.edge('A', 'B', label='Audio chunks')
    dot.edge('B', 'C')
    dot.edge('C', 'O', label='English Text')
    dot.edge('O', 'D', label='Refined Text')
    dot.edge('D', 'E', label='Indic Text')
    dot.edge('E', 'F', label='Indic Audio')

    dot.render('Project-Report-Template-3.9/Figures/architecture', view=False)
    print("Generated architecture diagram at Project-Report-Template-3.9/Figures/architecture.png")

if __name__ == '__main__':
    generate_diagram()
