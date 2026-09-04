> # ⛔ OBSOLETO — NÃO USE COMO RESULTADO ATUAL
>
> Arquivado em 4 set 2026. Esta rodada foi produzida **antes** de duas
> correções que mudam justamente o que ela mede:
>
> - **achado 07** — o `SYSTEM_PROMPT` daqui é a versão que proibia dizer
>   "não sei". As respostas interpretativas abaixo contêm a especulação que
>   essa correção eliminou. Elas documentam o defeito, não o comportamento.
> - **achado 06 / ambiguidade de departamento** — a tool estruturada daqui
>   somava departamentos homônimos em vez de relatar a ambiguidade. As
>   contagens abaixo herdam esse erro.
>
> O cabeçalho também é anterior ao carimbo de configuração: não registra
> `NUM_CTX`, `REASONING_EFFORT` nem o SHA do prompt, então não há como
> saber com que configuração exata cada célula foi gerada. E não existe
> `testes_pipelines.jsonl` correspondente — o registro bruto ainda não
> existia. **Nada aqui é comparável com uma rodada nova.**
>
> Mantido só como registro histórico do estado que motivou os sete achados.

# Comparação dos três pipelines — dados de docentes

Gerado por `interfaces/comparar.py` em 2026-09-03 17:47.

- Modelo LLM: `qwen2.5:32b-instruct-q4_K_M`
- Modelo de embedding: `BAAI/bge-m3` (dim 1024)
- TOP_K: 10

Os campos **Qualidade** e **Alucinou?** são para preenchimento manual —
o runner não julga resposta. **Fonte** é preenchida automaticamente e é
verificável: diz de onde o dado saiu de fato.

**Como julgar as perguntas objetivas:** o pipeline 2 consulta o SQLite
por template, sem LLM — ele é incapaz de alucinar, então a resposta dele
é a verdade-base contra a qual os pipelines 1 e 3 devem ser comparados.
Nas interpretativas não há verdade-base automática; aí o julgamento é todo
manual, conferindo contra os chunks citados no campo **Como**.

## Quantos professores tem o Departamento de Ciência da Computação?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: BRUNO JOSE DEMBOGURSKI, FILIPE BRAIDA DO CARMO, MARCIO NUNES DE MIRANDA, ANGEL RAMON SANCHEZ DELGADO, NILTON JOSE RIZZO, ANDRE LUIZ DE CASTRO LEAL, MARCELO SOBREIRO, ROBSON MARIANO DA SILVA, LUIZ PAULO CARVALHO DA SILVA, CLAUDIO CESAR SACCOMORI JUNIOR
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Os documentos mencionam dois docentes pertencentes ao Departamento de Ciência da Computação/IM:
> 
> 1. BRUNO JOSE DEMBOGURSKI.
> 2. FILIPE BRAIDA DO CARMO.
> 
> Portanto, com base nos documentos fornecidos, há 2 professores no Departamento de Ciência da Computação.

### Pipeline 2-estruturado

- **Fonte:** sqlite
- **Como:** departamento casado por fuzzy: 'DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM' (score 95)
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O departamento 'DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM' tem 6 docentes: BRUNO JOSE DEMBOGURSKI, FILIPE BRAIDA DO CARMO, NATALIA CHAVES LESSA, RICARDO CORDEIRO CORREA, RONALDO E SILVA VIEIRA, UBIRATAM CARVALHO DE PAULA JUNIOR

### Pipeline 3-agente

- **Fonte:** sqlite
- **Como:** tools chamadas: buscar_docentes_por_departamento
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O Departamento de Ciência da Computação conta com 6 professores. Eles são:
> 
> - Bruno Jose Dembogurski
> - Filipe Brada do Carmo
> - Natalia Chaves Lessa
> - Ricardo Cordeiro Corrêa
> - Ronaldo E Silva Vieira 
> - Ubiratam Carvalho de Paula Junior
> 
> É uma equipe que certamente contribui bastante para a formação dos alunos na área!

---

## Liste os docentes do Departamento de Matemática.

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: FELIPE LEITE COELHO DA SILVA, ANGELICA FRANCISCA DE ARAUJO, LUCIANO VIANNA FELIX, CLAUDIO CESAR SACCOMORI JUNIOR, ANDRES MAURICIO LOPEZ BARRAGAN, ANDRE LUIZ MARTINS PEREIRA, SERGIO DRUMOND VENTURA, DUILIO TADEU DA CONCEICAO JUNIOR, LUIS ALBERTO TOSCANO MEDRANO, EULINA COUTINHO SILVA DO NASCIMENTO
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Os docentes listados no Departamento de Matemática são:
> 
> - FELIPE LEITE COELHO DA SILVA
> - ANGELICA FRANCISCA DE ARAUJO
> - LUCIANO VIANNA FELIX
> - CLAUDIO CESAR SACCOMORI JUNIOR
> - ANDRES MAURICIO LOPEZ BARRAGAN
> - ANDRE LUIZ MARTINS PEREIRA
> - SERGIO DRUMOND VENTURA
> - DUILIO TADEU DA CONCEICAO JUNIOR
> - LUIS ALBERTO TOSCANO MEDRANO
> - EULINA COUTINHO SILVA DO NASCIMENTO

### Pipeline 2-estruturado

- **Fonte:** sqlite
- **Como:** departamento casado por fuzzy: 'DEPARTAMENTO DE MATEMÁTICA' (score 100)
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O departamento 'DEPARTAMENTO DE MATEMÁTICA' tem 29 docentes: ALINE MAURICIO BARBOSA, ANA LUIZA DA CONCEIÇÃO TENÓRIO, ANDRE LUIZ MARTINS PEREIRA, ANDRES MAURICIO LOPEZ BARRAGAN, ANGEL RAMON SANCHEZ DELGADO, ANGELICA FRANCISCA DE ARAUJO, ANTONIO CARLOS GONCALVES, CARLOS ANDRES REYNA VERA TUDELA, CELSO GUIMARAES BARBOSA, CLAUDIO CESAR SACCOMORI JUNIOR, DANIEL REIS DE OLIVEIRA, DIEGO DA SILVA BARROS, DUILIO TADEU DA CONCEICAO JUNIOR, EDILSON MARCELINO SILVA, EULINA COUTINHO SILVA DO NASCIMENTO, FELIPE LEITE COELHO DA SILVA, JOSIANE DA SILVA CORDEIRO COELHO, KARL IGOR MARTINS GUERRA, LEANDRO TOMAZ DE ARAUJO, LUCIANA CAVALCANTI DE MELO, LUCIANO VIANNA FELIX, LUIS ALBERTO TOSCANO MEDRANO, MONTAUBAN MOREIRA DE OLIVEIRA JUNIOR, OBERDAN MIGUEL RODRIGUES DE SOUZA, ORLANDO DOS SANTOS PEREIRA, RENAN DE SOUZA TEIXEIRA, SERGIO DRUMOND VENTURA, WELSON ANTONIO DE OLIVEIRA, WILIAN JERONIMO DOS SANTOS

### Pipeline 3-agente

- **Fonte:** sqlite
- **Como:** tools chamadas: buscar_docentes_por_departamento
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O Departamento de Matemática da UFRRJ tem um total de 29 docentes cadastrados. É um número bem significativo, não acha? Se você precisar de informações sobre algum docente específico, é só me dizer! Posso tentar te ajudar com nomes ou áreas de pesquisa.

---

## Quantos docentes tem o Departamento de Física?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: MARCELO AZEVEDO NEVES, DEBORA NUNES BARROS DE VASCONCELOS, DEBORA CARVALHO DE MELO RODRIGUES, LUIZ FILIPE DE OLIVEIRA GUIMARAES, MARCELO AZEVEDO NEVES, ARTUR JORGE DA SILVA LOPES, RICARDO JOSE SCHERER SANTOS, JOAO JOSE DOS SANTOS ALVES, IGOR TUCHE DE ALMEIDA DINIZ, ISAIAS GONZAGA DE OLIVEIRA
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> De acordo com os documentos fornecidos, há informações sobre nove docentes do Departamento de Física:
> 
> 1. DEBORA NUNES BARROS DE VASCONCELOS
> 2. DEBORA CARVALHO DE MELO RODRIGUES
> 3. LUIZ FILIPE DE OLIVEIRA GUIMARAES
> 4. MARCELO AZEVEDO NEVES
> 5. ARTUR JORGE DA SILVA LOPES
> 6. RICARDO JOSE SCHERER SANTOS
> 7. JOAO JOSE DOS SANTOS ALVES
> 8. IGOR TUCHE DE ALMEIDA DINIZ
> 9. ISAIAS GONZAGA DE OLIVEIRA
> 
> Portanto, o Departamento de Física tem nove docentes listados nos documentos fornecidos.

### Pipeline 2-estruturado

- **Fonte:** sqlite
- **Como:** departamento casado por fuzzy: 'DEPARTAMENTO DE FÍSICA' (score 100)
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O departamento 'DEPARTAMENTO DE FÍSICA' tem 14 docentes: ARTUR JORGE DA SILVA LOPES, CLAUDIO MAIA PORTO, CRESUS FONSECA DE LIMA GODINHO, DEBORA CARVALHO DE MELO RODRIGUES, DEBORA NUNES BARROS DE VASCONCELOS, IGOR TUCHE DE ALMEIDA DINIZ, ISAIAS GONZAGA DE OLIVEIRA, JOAO JOSE DOS SANTOS ALVES, LUCAS MODESTO DA COSTA, LUIZ FILIPE DE OLIVEIRA GUIMARAES, MARCELO AZEVEDO NEVES, MARCOS CARDOSO RODRIGUEZ, RICARDO JOSE SCHERER SANTOS, RODRIGO DE SOUSA GONCALVES

### Pipeline 3-agente

- **Fonte:** sqlite
- **Como:** tools chamadas: buscar_docentes_por_departamento
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> O Departamento de Física da UFRRJ conta atualmente com 22 docentes cadastrados. Se precisar de mais alguma informação, é só pedir!

---

## Quais docentes pesquisam sobre Inteligência Artificial?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: ANA LISA NISHIO, ANA LISA NISHIO, CHRISTIANE AGUIAR DE ANDRADE, RICARDO CORDEIRO CORREA, DORA SORAIA KINDEL, BRUNO ARAUJO FURTADO DE MENDONCA, IDIO ALVES DE SOUSA FILHO, RONALDO MALHEIROS GREGORIO, ALDENILSON DOS SANTOS VITORINO COSTA, ALDENILSON DOS SANTOS VITORINO COSTA
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Com base nos documentos fornecidos, não há menção explícita de nenhum dos docentes pesquisando diretamente sobre Inteligência Artificial. No entanto, Christiane Aguiar de Andrade integra o Núcleo Docente Estruturante do curso de Administração e participa da elaboração de diretrizes para o uso de Inteligência Artificial na formação dos estudantes, embora a pesquisa específica em Inteligência Artificial não seja mencionada como um foco principal.

### Pipeline 2-estruturado

- **Fonte:** nenhuma
- **Como:** melhor score fuzzy 55 < limiar 70
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Não consegui identificar um departamento na pergunta — este pipeline só responde a perguntas objetivas sobre departamentos.

### Pipeline 3-agente

- **Fonte:** chromadb
- **Como:** tools chamadas: busca_vetorial_sigaa
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Baseado na informação que encontrei, alguns docentes da UFRRJ pesquisam sobre Inteligência Artificial:
> 
> 1. **NATALIA CHAVES LESSA** - Trabalha no Departamento de Ciência da Computação do Instituto Multidisciplinar (IM). Embora não tenha sido especificamente mencionado que ela pesquisa em Inteligência Artificial, é possível que ela esteja envolvida nessa área dada sua experiência em Engenharia de Sistemas e Computação.
> 
> 2. **Docente não nomeado** - Um dos docentes pesquisadores tem experiência na área de Desenvolvimento de Sistemas e atua nas áreas de Inteligência Artificial, Mineração de Dados e Sistemas de Recomendação.
> 
> Para obter informações mais detalhadas ou atualizadas, é recomendável verificar o site do professor ou entrar em contato diretamente com o departamento de interesse.

---

## Algum professor trabalha com banco de dados ou ciência de dados?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: BRUNO D ASSIS ROCHA, ADEILTON PEDRO DE ALCANTARA, ANTONIO CARLOS GONCALVES, RICARDO CORDEIRO CORREA, RODRIGO CARLOS MARQUES PEREIRA, FILIPE BRAIDA DO CARMO, ELTON CARNEIRO MARINHO, MAICON CLAUDIO DA SILVA, SERGIO CORREIA BARBOSA, RODOLFO ROCHA DOS SANTOS
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Baseado nos documentos fornecidos, não há menção específica sobre nenhum dos docentes que trabalhe diretamente com bancos de dados ou ciência de dados como foco principal. No entanto, Adeilton Pedro de Alcântara possui conhecimentos em ferramentas como R e SAS, que são frequentemente utilizadas na ciência de dados para análise estatística. Além disso, Elton Carneiro Marinho tem uma formação voltada à arquitetura baseada em Blockchain, o que pode ter algum relacionamento com bancos de dados, dependendo do contexto específico da sua pesquisa e atividades acadêmicas. Entretanto, essas menções são indiretas e não indicam um foco explícito em banco de dados ou ciência de dados nos perfis apresentados.

### Pipeline 2-estruturado

- **Fonte:** nenhuma
- **Como:** melhor score fuzzy 56 < limiar 70
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Não consegui identificar um departamento na pergunta — este pipeline só responde a perguntas objetivas sobre departamentos.

### Pipeline 3-agente

- **Fonte:** chromadb
- **Como:** tools chamadas: busca_vetorial_sigaa
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Encontrei um docente que tem experiência relevante com análise de dados no contexto da ciência:
> 
> **Docente: ADEILTON PEDRO DE ALCANTARA**
> - **Departamento:** Departamento de Tecnologias e Linguagens/IM
> - **Formação e Experiência:** Tem conhecimentos em várias ferramentas de análise de dados, incluindo R, WinBUGS, SPSS e SAS. Além disso, possui graduação em Estatística, mestrado em Estudos Populacionais e Pesquisas Sociais, e doutorado em Estatística pela Universidade Estadual de Campinas.
> 
> Parece que o professor Adeilton Pedro de Alcântara é uma boa opção para quem busca por conhecimento sobre ciência de dados! Se precisar de mais detalhes ou tiver outras dúvidas, estou à disposição.

---

## Que docente tem formação em engenharia?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: ELISABETE DO NASCIMENTO ROCHA, ELISABETE DO NASCIMENTO ROCHA, DANIEL FONSECA DE CARVALHO, ALESSANDRA CARREIRO BAPTISTA, JOSE LUCENA BARBOSA JUNIOR, ALESSANDRA CARREIRO BAPTISTA, CLAUDIA DAZA ANDRADE, HENRIQUE VIEIRA DE MENDONCA, JOAO PAULO BARRETO CUNHA, MAURICIO CORDEIRO MANCINI
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Todos os docentes mencionados têm formação em alguma área de engenharia. No entanto, para responder especificamente com base nos documentos fornecidos:
> 
> - DANIEL FONSECA DE CARVALHO: Formado em Engenharia Agrícola pela ESAL/UFLA.
> - JOSE LUCENA BARBOSA JUNIOR: Graduado em Engenharia Química pela UFRRJ.
> - ALESSANDRA CARREIRO BAPTISTA: Graduação em Engenharia de Agrimensura pela Universidade Federal de Viçosa.
> - CLAUDIA DAZA ANDRADE: Formada em Engenheira Civil.
> - HENRIQUE VIEIRA DE MENDONCA: Engenheiro Agrícola e Ambiental.
> - JOAO PAULO BARRETO CUNHA: Possui mestrado e doutorado na área de engenharia agrícola - mecanização.
>   
> Portanto, todos os docentes listados têm formação em engenharia.

### Pipeline 2-estruturado

- **Fonte:** nenhuma
- **Como:** melhor score fuzzy 68 < limiar 70
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Não consegui identificar um departamento na pergunta — este pipeline só responde a perguntas objetivas sobre departamentos.

### Pipeline 3-agente

- **Fonte:** chromadb
- **Como:** tools chamadas: busca_vetorial_sigaa
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Temos vários docentes com formação em engenharia na UFRRJ, aqui estão alguns exemplos:
> 
> - **ELISABETE DO NASCIMENTO ROCHA**, do Departamento de Geociências. Apesar da sua formação principal ser em geologia, ela possui sólida experiência e atua em disciplinas relacionadas a engenharia.
>   
> - **DANIEL FONSECA DE CARVALHO**, do Departamento de Engenharia. Ele é Engenheiro Agrícola com mestrado e doutorado na mesma área.
> 
> - **ALESSANDRA CARREIRO BAPTISTA** também pertence ao Departamento de Engenharia. Ela possui formação em Engenharia de Agrimensura, além de mestrado e doutorado em Engenharia Civil.
> 
> - **CLAUDIA DAZA ANDRADE**, do mesmo departamento, é Engenheira Civil com mestrado e doutorado na área de Recursos Hídricos.
> 
> Esses são apenas alguns exemplos. A UFRRJ tem outros docentes que atuam em diferentes áreas da engenharia, como você pode ver nos perfis dos demais docentes mencionados.

---

## Qual o telefone pessoal do reitor da UFRRJ?

### Pipeline 1-vetorial

- **Fonte:** chromadb
- **Como:** 10 chunks recuperados: CARLOS AUGUSTO BRANDAO DE CARVALHO, LEONARDO DUARTE BATISTA DA SILVA, TIAGO CRUZ DE FRANCA, MARCIO REIS PEREIRA DE SOUSA, RICARDO LUIZ LOURO BERBARA, MARCELLO VINICIUS DORIA CALVOSA, RODRIGO CARLOS MARQUES PEREIRA, RICARDO LUIZ LOURO BERBARA, MARCELO AZEVEDO NEVES, LUCAS MODESTO DA COSTA
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Não encontrou. Os documentos fornecidos não contêm o telefone pessoal do Reitor da UFRRJ.

### Pipeline 2-estruturado

- **Fonte:** nenhuma
- **Como:** melhor score fuzzy 48 < limiar 70
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Não consegui identificar um departamento na pergunta — este pipeline só responde a perguntas objetivas sobre departamentos.

### Pipeline 3-agente

- **Fonte:** nenhuma
- **Como:** nenhuma tool chamada — o LLM respondeu de cabeça
- **Qualidade:** _(preencher)_
- **Alucinou?:** _(preencher)_

> Desculpe, mas não posso ajudar com isso. Não é apropriado fornecer informações pessoais de nenhum funcionário, incluindo o reitor. Para contatos oficiais relacionados à administração da universidade, você pode verificar o site oficial da UFRRJ ou entrar em contato com a secretaria.

---
