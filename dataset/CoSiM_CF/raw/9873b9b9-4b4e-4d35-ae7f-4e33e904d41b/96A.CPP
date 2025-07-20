#include <cstring>
#include <iostream>
#include <stdio.h>
using namespace std;
bool check(int index, char word[]) {
  int count = 0;
  char chk = word[index];
  for (int i = index; strlen(word); i++) {
    if (chk == word[i]) {
      count++;
    }
  }
  if (count >= 7)
    return true;
}

int main() {
  char str[101];
  bool res;
  int flag = 0;
  scanf("%s", str);
  for (int i = 0; i < strlen(str) - 7; i++) {
    res = check(i, str);
    if (res == true)
      flag = 1;
  }
  if (flag == 1)
    printf("YES");
  else
    printf("NO");
  return 0;
}
