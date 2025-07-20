#include <cstdio>
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
  char word[1001];
  scanf("%s", word);
  word[0] = toupper(word[0]);
  printf("%s\n", word);
  return 0;
}
