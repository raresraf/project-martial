// Codeforces
// 231A - Team

#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
  int numProblems = 0;
  int a = 0, b = 0, c = 0, total = 0;

  scanf("%d\n", &numProblems);

  while (numProblems--) {
    scanf("%d %d %d\n", &a, &b, &c);
    if (a + b + c >= 2)
      total++;
  }

  printf("%d\n", total);

  return 0;
}
