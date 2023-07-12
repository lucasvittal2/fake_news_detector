# Detecção de Fake News Com DeepLearning 📰🎭🧠



### Introdução

Num mundo onde cada vez mais tem fluxo de informação, cada vez mais tem-se visto a necessidade de averiguar se uma informação é falsa ou verdadeira.  Pessoas que lidam com jornalismo no seu dia a dia e até mesmo pessoas minimamente informadas conseguem detectar com relativa facilidade uma fake news, contudo em agluns casos nem mesmo esse tipo de pessoa consegue detectar fazendo com que o falso seja entendido como verdadeiro. Se até mesmo especialista e pessoas informada tem dificuldaes para determinar se uma informação é falsa ou verdadeira havemos de convir que muito meno a opnião publica no geral, e justamente por isso uma fake news pode ter efeitos tão destrutivos na sociedade, pois mentir e fazer as pessoas acreditar numa informação é muito mais fácil do que desmenti-la, desmentir uma informação falsa requer muito mais esforço do que  espalhar uma e quanto maior for o tempo que  se elva para fazer isso mais difícil será comcertar os estragos causados pela noticias falsas..

Pensando nisso, várias entidades jornalísticas vem empenhando esforços para detectar e desmentir fake new, um procedimento normalmente realizado por um especialista na área. O grande problema é que por questões de escalabilidade não é possivel que um espeecialista avalie uma grande quantidade de fake news e o procedimento de avaliação  de uma notícia e a publicação de sua validade requer um tempo considerável ao ser feita por um ser humano, sendo que um noticia falsa muitas das vezes é divulgada por robôs de disparo em massa. Sendo assim é necessário haver uma solução que tanto possa avaliar se uma notícia é falsa ou verdadeira como também o pssa fazer de forma o mais rápida possivel e de forma automatizada.

Justamente considerando-se essa demanda cogitou-se realizar este projeto, aonde iremos propor uma solução que envolve processamento de linguagem natural e DeepLearning para avaliar se uma otícia é falsa ou verdadeira.

Como um primeiro MVP deste projeto pensou-se  em criar um modelo de Deep Learning que a partir do banco de dados da Uniersity of Victoria( Universidade de tecnologia do Canadá)  se pudesse prever se um noticia era falsa ou verdadeira.

O Daset utilizado possui as seguintes caracterpisticas:

- title : título da notícia
- text : Conteúdo textual da notícia
- date : data em que a notícia foi publicada
- subject : tema abordado pela notícia

Esse dataset fora extraído da seguinte forma:

 - As notícias verdadeiras foram extraidas de artigo do site reuters.com, um site de notícias mundial cuja informação é confiável 
 - As notícias falsas foram extraídas dos mais diversas fontes e de artigos aos quais não são confiáveis.
 - Todos os artigos extraídos foram coletados entre os anos de 2016 e 2017
 - Erros textuais e gramaticais foram mantidos nas fake news, preservando seu formato original

O dataset não explíccita o label das notícias por meio de uma varável, contudo ele é constituido de dois  arquivos csv rotulados (`true.csv`, `fake.csv`), onde posteriormente criamos os labels para cada um desse arquivos e  depois os concatenando e embaralhando os dados.

Mais informações e detalhes sobre o dateset utilizdo neste projeto pode ser enontrado no link abaixo:

[Fake_News_Dataset_ReadMe](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf)

 O link para acesso e download do dataset pode ser encontrado no próprio site da Unniversity of Victoria:

[fake-news-detection-datasets](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)


Já a pasta do projeto com todos os arquivos e dependências pode ser acessada abaixo:

[Fake News Dectetion Project Folder](https://drive.google.com/drive/folders/11cZBjo4KEdqeFi59E_GlDIthiU77zxP5?usp=sharing)

O notebook encontra-se dentro da pasta Code.

