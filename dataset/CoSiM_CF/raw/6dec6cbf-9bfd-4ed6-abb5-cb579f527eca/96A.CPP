#include <bits/stdc++.h>
using namespace std;
int main() {
  bool danger = false;
  char field[101];
  scanf("%s", &field);

  int last = (int)field[0] - 48;
  int howMuch = 1;
  for (int i = 1; i < strlen(field); i++) {
    if (last == (int)field[i] - 48) {
      howMuch += 1;
      if (howMuch >= 7)
        danger = true;
    } else {
      howMuch = 1;
      last = (int)field[i] - 48;
    }
  }
  if (danger)
    printf("YES\n");
  else
    printf("NO\n");
}