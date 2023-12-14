# InvisiMark

## Como Executar o Projeto?

### Adicionando as Pastas Necessárias

1. Crie uma pasta chamada `images` no diretório raiz do projeto.
2. Dentro da pasta `images`, crie as seguintes subpastas:
   - `extracted_watermark`
   - `marked_images`
   - `watermarks`
3. Adicione uma pasta chamada `data` no diretório raiz e crie um arquivo chamado `users.json` dentro dela. No arquivo JSON recém-criado, adicione um array vazio `[]`.

### Instalando Dependências

Todas as dependências do projeto estão listadas no arquivo `requirements.txt` (Python) ou `package.json` (Node.js). Para instalar as dependências, execute os seguintes comandos:

```bash
pip install -r requirements.txt
```

```bash
npm i
```

### Executando o projeto

Existem dois comandos principais para executar o projeto:

```bash
flask run
```

Este comando inicia o aplicativo Flask.

```bash
npm run tailwind
```

Este comando carrega os estilos CSS da aplicação.

### Considerações finais

- Após seguir esses passos, basta criar um cadastro e começar a usar a aplicação. Não se esqueça de cadastrar marcas d'água para utilizar na aplicação.
- Todos os métodos de inserção e extração de marcas d'água estão localizados na pasta `invisimark/services`.
