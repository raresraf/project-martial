#include <stdio.h>
#include <string>
using namespace std;

int main() {
  string hello = "hello";
  char word[101];
  bool temp = false;
  int i = 0;
  scanf("%100s", word);
  for (char &a : word) {
    if (a == hello[i]) {
      i++;
      if (i == 5)
        break;
    }
  }

  if (i == 5)
    printf("YES");
  else
    printf("NO");

  return 0;
}
