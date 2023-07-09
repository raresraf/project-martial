# Bash Hacks to make it work 

## Download stuff

```
for i in {1..78}; do; echo "curl -OL https://github.com/microsoft/vscode/archive/refs/tags/1.$i.0.tar.gz" | bash -; done
```

## Mass untar

```
for f in *.tar.gz; do tar xf "$f"; done
```

## Get all eslint disablements

```
find . -name "*.ts" -print0 | xargs -0 -I file cat file > merged.file && cat merged.file | grep "// eslint" | sed -e 's/^[[:space:]]*//' | sed 's/.*\/\///' | sed -e 's/^[[:space:]]*//'
```

## Get all eslint disablements and count 

```
find . -name "*.ts" -print0 | xargs -0 -I file cat file > merged.file && cat merged.file | grep "// eslint" | sed -e 's/^[[:space:]]*//' | sed 's/.*\/\///' | sed -e 's/^[[:space:]]*//' | tr " " "\n" | sed 's/,*$//g' | sort -n | uniq -c
```

## All toghether

```
for i in {1..78}; do; cd "/Users/raf/code/test/vscode-1.$i.0" && sleep 1 && find . -name "*.ts" -print0 | xargs -0 -I file cat file > merged.file && cat merged.file | grep "// eslint" | sed -e 's/^[[:space:]]*//' | sed 's/.*\/\///' | sed -e 's/^[[:space:]]*//' | tr " " "\n" | sed 's/,*$//g' | sort -n | uniq -c > "/Users/raf/code/results/vscode-1.$i.0"; done;
```
