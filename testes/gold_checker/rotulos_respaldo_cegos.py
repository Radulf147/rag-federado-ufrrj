"""
Rótulos CEGOS de respaldo — 20 perfis, feitos à mão ANTES do classificador.

PROTOCOLO
    - Amostra sorteada com `random.seed(20260905)`, `random.sample(sorted(...), 20)`
      sobre os 69 docentes distintos citados nas 21 respostas de atribuição.
      A amostra não foi escolhida: foi sorteada, e a semente está aqui.
    - Rotulados olhando SÓ o perfil e o tema. Sem ver qual resposta citou cada
      um, sem ver o veredito automático — que nesta data ainda não existia.
    - Este arquivo é commitado ANTES do classificador. A ordem no histórico do
      git é a evidência da cegueira; sem ela, "foi cego" é palavra minha.

COMPOSIÇÃO DA AMOSTRA — reportada porque ela condiciona a concordância
    10 perfis apenas institucionais (nome, departamento, contato)
    10 perfis substantivos (com Perfil, Formação ou Áreas de interesse)

    A metade institucional é a parte FÁCIL: qualquer classificador acerta, porque
    não há texto onde procurar. Concordância alta puxada por ela não significa
    nada. **O que vale é a concordância nos 10 substantivos**, onde a decisão
    entre COM RESPALDO e INCONCLUSIVO exige julgar conteúdo.

UMA CORREÇÃO FEITA DURANTE A ROTULAGEM, registrada
    O primeiro dump truncava o perfil em 520 caracteres. MARCOS BACIS CEDDIA
    ficaria INCONCLUSIVO por isso — a palavra "agroecologia" aparece por volta do
    caractere 1100, dentro de "física do solo, geoestatística, agroecologia e
    agricultura digital". Reli os três limítrofes por inteiro antes de fixar o
    rótulo. Ler MAIS do perfil não fere a cegueira; ler menos é que produz rótulo
    errado — foi assim que errei amb-04 e amb-06.
"""

SEMENTE = 20260905

SEM_RESPALDO, COM_RESPALDO, INCONCLUSIVO = "SEM_RESPALDO", "COM_RESPALDO", "INCONCLUSIVO"

ROTULOS = [
    # --- perfis apenas institucionais: não há onde procurar o tema -----------
    ("ALESSANDRA DE ANDRADE RINALDI", "movimentos sociais", SEM_RESPALDO,
     "só Telefone e E-mail"),
    ("ANNA MARIA PEREIRA ESTEVES", "movimentos sociais", SEM_RESPALDO,
     "só Telefone e E-mail"),
    ("BRUNA MOTTA DOS SANTOS", "movimentos sociais", SEM_RESPALDO,
     "Lattes 'link não informado', Telefone, E-mail"),
    ("BRUNO JOSE DEMBOGURSKI", "inteligencia artificial", SEM_RESPALDO,
     "só Telefone e E-mail"),
    ("JAIME RODRIGO DA SILVA MIRANDA", "movimentos sociais", SEM_RESPALDO,
     "só Telefone e E-mail"),
    ("MAGDA GISELA CRUZ DOS SANTOS", "movimentos sociais", SEM_RESPALDO,
     "só E-mail"),
    ("MARCELO DA COSTA MACIEL", "movimentos sociais", SEM_RESPALDO,
     "só Telefone e E-mail"),
    ("MAURICIO HOELZ VEIGA JUNIOR", "movimentos sociais", SEM_RESPALDO,
     "Lattes, Telefone, E-mail"),
    ("RAPHAEL CASTELO BRANCO DA SILVA", "movimentos sociais", SEM_RESPALDO,
     "Lattes 'link não informado', E-mail"),
    ("TAMIS PORFIRIO COSTA CRISOSTOMO RAMOS", "movimentos sociais", SEM_RESPALDO,
     "só Telefone e E-mail"),

    # --- perfis substantivos COM evidência do tema --------------------------
    ("ADRIANA OLIVEIRA ANDRADE", "estatistica", COM_RESPALDO,
     "Perfil: 'Professora Adjunta de Estatística'; Áreas: 'Estatística Aplicada'"),
    ("ANTONIO CARLOS GONCALVES", "estatistica", COM_RESPALDO,
     "Perfil: 'Há mais de 30 anos atuando na área da Estatística'; Áreas: 'Ensino de Estatística'"),
    ("CHRISTIAN MARIE VICTOR SIMON DUTILLEUX", "literatura", COM_RESPALDO,
     "Perfil: 'Professor de Teoria Literária e Literatura Comparada desde 2013'"),
    ("LUIS CARLOS ALVES DE MELO", "literatura", COM_RESPALDO,
     "Formação: 'Doutor em Letras (Teoria da Literatura e Literatura Comparada)'"),
    ("MARCOS BACIS CEDDIA", "agroecologia", COM_RESPALDO,
     "Perfil: 'física do solo, geoestatística, agroecologia e agricultura digital' (~char 1100)"),
    ("RONALDO E SILVA VIEIRA", "inteligencia artificial", COM_RESPALDO,
     "Áreas: 'Jogos, Inteligência Artificial, Ciência de Dados'"),
    ("ROSEMARY GONCALO AFONSO", "literatura", COM_RESPALDO,
     "Formação: 'Doutora em Literatura Portuguesa'"),
    ("VALERIA ROSITO FERREIRA", "literatura", COM_RESPALDO,
     "Formação: 'DOUTORADO EM LITERATURA COMPARADA'; Áreas: 'LITERATURA COMPARADA'"),

    # --- perfis substantivos SEM evidência: inconclusivo, não falha ---------
    ("ADRIA RAMOS DE LYRA", "inteligencia artificial", INCONCLUSIVO,
     "perfil completo (296 chars): Formação 'Doutorado em Computação', Áreas "
     "'Ciência da Computação, Humanidades Digitais'. Substantivo, e nada sobre IA. "
     "Pode ser que pesquise e não tenha escrito — por isso não é falha."),
    ("HENRIQUE VIEIRA DE MENDONCA", "agroecologia", INCONCLUSIVO,
     "perfil de 978 chars, lido inteiro: tratamento de resíduos, microalgas, "
     "bioenergia, wetlands. Áreas: 'Controle da poluição, tratamento de águas "
     "residuárias, bioenergia e microalgas'. Nenhuma menção a agroecologia. "
     "O próprio agente disse 'podem ter conexões indiretas' — é exatamente o "
     "caso em que o casamento de palavra não decide."),
]

POR_NOME = {nome: rotulo for nome, _tema, rotulo, _porque in ROTULOS}
TEMA_POR_NOME = {nome: tema for nome, tema, _rotulo, _porque in ROTULOS}

DISTRIBUICAO = {SEM_RESPALDO: 10, COM_RESPALDO: 8, INCONCLUSIVO: 2}
