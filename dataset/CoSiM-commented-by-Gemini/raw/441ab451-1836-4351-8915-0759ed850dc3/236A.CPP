#include <cstdio>

using namespace std;

int check[26];

int main() {
  int n = 0, i, temp;
  char x[101];
  scanf("%s", &x);
  for (i = 0; x[i] != 0; i++) {
    temp = x[i] - 'a';
    if (check[temp] == 1) {
      continue;
    } else {
      check[temp] = 1;
      n++;
    }
  }
  if (n % 2 == 0)
    printf("CHAT WITH HER!");
  else
    printf("IGNORE HIM!");
  return 0;
}