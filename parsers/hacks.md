# Bash Hacks to make it work 

## Download stuff

```
for i in {1..78}; do; echo "curl -OL https://github.com/microsoft/vscode/archive/refs/tags/1.$i.0.tar.gz" | bash -; done
```

## Get all eslint disablements

```
find . -name "*.ts" -print0 | xargs -0 -I file cat file > merged.file; cat merged.file| grep "// eslint" | sed -e 's/^[[:space:]]*//' | sed 's/.*\/\///' | sed -e 's/^[[:space:]]*//'
```
