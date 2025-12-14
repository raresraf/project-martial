// Codeforces
// 282A - Bit++

#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
  int numStatements = 0;
  scanf("%d\n", &numStatements);

  string s = "";
  int total = 0;
  while (numStatements--) {
    getline(cin, s);
    if (s[1] == '+')
      total++;
    else
      total--;
  }

  printf("%d\n", total);

  return 0;
}
