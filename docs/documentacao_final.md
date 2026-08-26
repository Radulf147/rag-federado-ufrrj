1. Introdução e Objetivos

Sistemas acadêmicos universitários, como o Sistema Integrado de Gestão de Atividades Acadêmicas (SIGAA), concentram um vasto volume de dados cruciais para a comunidade discente e docente. Contudo, a navegação fragmentada e a arquitetura de informação complexa frequentemente resultam em desinformação, dificultando o acesso rápido a dados como contatos de docentes, ementas de disciplinas, editais de bolsas e serviços institucionais.

Para mitigar este problema, este trabalho propõe o desenvolvimento de um Agente RAG (Retrieval-Augmented Generation) Federado. A solução visa acoplar a capacidade de compreensão de linguagem natural dos Grandes Modelos de Linguagem (LLMs) a uma base de conhecimento atualizada e extraída dinamicamente do SIGAA.

O objetivo principal é construir uma arquitetura descentralizada, baseada na rede Mastodon (Fediverso), onde um agente autônomo atue como um facilitador de informações (semelhante ao assistente "Grok" da rede X/Twitter). Para garantir a precisão das respostas e mitigar o fenômeno da alucinação (geração de informações falsas pelo LLM), o projeto evolui do paradigma de Naive RAG para uma abordagem de Agentic RAG com Armazenamento Híbrido, separando consultas semânticas de consultas determinísticas.

2. Arquitetura do Sistema e Metodologia

A arquitetura do sistema foi projetada sob os princípios de modularidade, desacoplamento e escalabilidade. O ciclo de vida dos dados e a inferência do modelo estão divididos em três módulos principais.

2.1. Módulo 1: Pipeline ETL e Armazenamento Híbrido

O pipeline de Extração, Transformação e Carga (ETL) é responsável por manter a base de conhecimento do agente sincronizada com o SIGAA. A extração (Scraping) ocorre de forma assíncrona, navegando em múltiplos níveis hierárquicos (ex.: varredura de departamentos seguida da coleta de perfis individuais de docentes). O texto extraído passa por um processo de Chunking estrutural (divisão por sentenças com sobreposição) e, em seguida, é vetorizado através de modelos da arquitetura Transformer.

Para suportar a complexidade das consultas futuras, a persistência dos dados adota uma Arquitetura Híbrida:

Vector Store (ChromaDB): Armazena os embeddings de textos longos e descritivos (ex.: biografias, ementas, áreas de pesquisa). Otimizado para cálculo de distância vetorial e busca de similaridade semântica.

Document Store Genérico (SQLite): Dados estruturados que exigem determinismo e precisão absoluta (ex.: a lista exata de docentes de um departamento ou diretórios de serviços) são salvos no formato JSON (schema-less) numa tabela genérica relacional.

2.2. Módulo 2: Motor de Inferência (Agentic RAG)

Neste módulo, ocorre a interação com o usuário. Utilizando o framework Haystack 2.x, o LLM local (Mistral, servido via Ollama) não atua apenas como um gerador de texto, mas como um Agente Autônomo. Através da técnica de Tool Calling (Function Calling), o modelo é equipado com ferramentas sistêmicas. Em tempo real, o LLM interpreta a intenção da pergunta do usuário e decide dinamicamente:

Se deve formular e executar uma query (consulta) na base SQLite para responder a perguntas exatas.

Se deve acionar o motor de busca vetorial no ChromaDB para responder a perguntas interpretativas ou conceituais.

2.3. Módulo 3: Integração Federada Descentralizada

Como camada de interface, o sistema transcende as interfaces web tradicionais e integra-se à rede Mastodon via ActivityPub. O agente opera como um bot escutando menções na rede. Ao ser acionado por um aluno de qualquer instância conectada à UFRRJ, o conteúdo do post (toot) atua como prompt. A resposta gerada pelo Módulo 2 é então publicada em resposta (thread), democratizando o acesso à informação em um ecossistema descentralizado.

2.4. Decisões de Design: O Paradoxo da Estruturação

Durante o desenvolvimento do Módulo 2, desafios inerentes aos sistemas RAG tradicionais exigiram a adoção de soluções de engenharia avançadas, consolidando o abandono de técnicas rudimentares em favor do estado da arte.

Por que o banco vetorial (ChromaDB) não é suficiente sozinho?

A premissa básica do RAG clássico (Naive RAG) é buscar os $K$ fragmentos de texto (chunks) cujos vetores são mais próximos matematicamente à pergunta do usuário. Contudo, essa abordagem de Busca Semântica é fundamentalmente ineficiente para agregações exaustivas e contagens.

Se um usuário solicitar "Liste todos os professores do Departamento de Computação", o banco vetorial não retornará a totalidade do departamento, mas apenas um subconjunto limitado pelo hiperparâmetro top_k. Aumentar o top_k artificialmente para tentar englobar todos os registros gera um novo problema: o esgotamento da Janela de Contexto do LLM. Enviar dezenas de milhares de tokens ao modelo resulta em latência extrema, estouro de memória (VRAM) e no fenômeno conhecido como Lost in the Middle — onde o modelo sofre "amnésia seletiva" e ignora informações localizadas no meio do prompt.

A solução arquitetural foi o acoplamento de um Document Store determinístico (SQLite). Ao persistir dados de listagem como JSONs brutos (schema-less) no SQLite, o sistema consegue realizar operações de contagem (COUNT) e filtragem com 100% de precisão e custo computacional quase nulo, enviando ao LLM apenas o resultado consolidado e mitigando o risco de alucinação.

Por que abandonamos Expressões Regulares (Regex) em favor de Tool Calling?

No Produto Mínimo Viável (MVP), o roteamento entre a busca vetorial (ChromaDB) e a busca exata era controlado por Expressões Regulares (Regex), como r"(todos os|lista de) professores". Embora funcional em escopos pequenos, essa abordagem apresentou-se insustentável a longo prazo.

A linguagem natural humana é infinitamente variável. Mapear manualmente todas as formas que um aluno poderia formular uma pergunta sobre turmas, disciplinas, calendários e projetos resultaria num código frágil e acoplado.

O abandono das Regex em prol do Tool Calling (Function Calling) representa a transição para o paradigma Agentic RAG. Ao registrar funções Python (como buscar_disciplinas_sqlite ou buscar_biografia_chroma) como ferramentas no Haystack, delega-se a interpretação da intenção ao próprio LLM. O modelo avalia a semântica da pergunta e decide autonomamente qual ferramenta invocar e quais parâmetros passar. Essa decisão garante escalabilidade irrestrita: novos scrapers do SIGAA podem ser adicionados ao sistema bastando acoplar uma nova ferramenta ao agente, sem a necessidade de reescrever lógicas de controle.

3. Stack Tecnológica

O ecossistema do projeto foi construído utilizando tecnologias open-source, priorizando a modularidade, o processamento assíncrono e a privacidade dos dados da universidade.

Haystack 2.x: Framework principal de orquestração do pipeline de Inteligência Artificial. Selecionado pela sua arquitetura moderna orientada a componentes, excelente suporte a Agentic RAG e Tool Calling.

ChromaDB: Banco de dados vetorial open-source. Escolhido por permitir armazenamento persistente local sem depender de APIs proprietárias em nuvem, garantindo baixa latência na recuperação de embeddings.

SQLite: Banco de dados relacional leve e embutido. Utilizado como Document Store genérico graças ao seu suporte nativo robusto a funções JSON (como json_extract), permitindo armazenamento determinístico schema-less.

Ollama (Mistral): Servidor de inferência local para Grandes Modelos de Linguagem (LLMs). Permite a execução de modelos avançados (como o Mistral) de forma parametrizada, isolada e sem custos de API externa, mantendo os dados da instituição privados.

BeautifulSoup4 & Asyncio/Httpx: Combinação utilizada na camada de extração (ETL). Enquanto o BeautifulSoup oferece parsing tolerante a falhas no HTML obsoleto do SIGAA, o httpx aliado ao asyncio (coroutines) permite concorrência em I/O, reduzindo drasticamente o tempo de scraping.

Docker & Docker Compose: Infraestrutura como código (IaC). Garante a reprodutibilidade do ambiente, isolando o banco de dados, o orquestrador ETL e o Agente em contêineres separados, além de facilitar o pass-through de recursos de hardware (GPU NVIDIA) para aceleração de matrizes.

Mastodon API: Protocolo ActivityPub utilizado para a federação da aplicação, permitindo a interfaceamento via menções em uma rede social livre.

4. Guia de Operação e Deploy (DevOps)

Toda a complexidade de orquestração do ambiente foi abstraída através de contêineres Docker e controlada por um shell script utilitário (rag.sh), garantindo facilidade na implantação em servidores institucionais ou máquinas locais.

Pré-requisitos: Docker, plugin Docker Compose V2 e, opcionalmente, Nvidia Container Toolkit para aceleração de GPU.

4.1. Construção da Imagem

O comando de build constrói a imagem principal, resolvendo as dependências listadas no requirements.txt sem executar o código:

./rag.sh build


4.2. Execução do Pipeline ETL

O comando abaixo inicializa o banco vetorial (ChromaDB) em plano de fundo (background) e aciona o contêiner efêmero do ETL. O pipeline raspa os dados do SIGAA de forma assíncrona, gera os embeddings e persiste os dados em volume de disco local, encerrando o contêiner automaticamente ao término.

./rag.sh etl


4.3. Inicialização do Agente de Inferência

Com a base de dados populada (Módulo 1 concluído), o ambiente interativo de perguntas e respostas é iniciado acoplando-se ao servidor Ollama hospedeiro:

./rag.sh agente


4.4. Gerenciamento e Limpeza

Para fins de manutenção e reestruturação da base de conhecimento, o ambiente possui um comando de teardown que paralisa os serviços e purga os volumes de dados locais de forma segura:

./rag.sh limpar
