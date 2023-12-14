# InvisiMark

## Como executar o projeto ?

### Adicionando pastas necessárias

Adicione uma pasta chamada `images` no projeto e dentro adicionar as seguintes três pastas:
- marked_images
- original_images
- watermarks

Após isso adicionar uma pasta chamada `data` e dentro dela uma arquivo chamada `users.json` e no arquivo json criado, adicionem um array vazio `[]`.

### Instalando dependências

Todas as dependências do projeto estão em um arquivo chamada requirements.txt ou package.json, para adicionar as dependências execute os seguintes comandos:

```bash
pip install -r requirements.txt
```
```bash
npm i
```

### Execute o projeto

Temos dois comandos para executar o projeto, o primeiro é o mais importante:

```bash
flask run
```

Pois com ele executamos nosso aplicativo, já o segundo é apenas para carregas nos styles css da aplicação:

```bash
npm run tailwind
```

### Considerações finais

- Com esses passos feitos só é preciso que você faça um cadastro e utilize a aplicação. Lembre-se de cadastrar marcas d'água para a aplicação.
- Todos os métodos de inserção e extração de marcas d'água se encontram na pasta `invisimark/services`
