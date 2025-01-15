#include <bits/stdc++.h>

using namespace std;
int main() {
  char chara[101];
  int total_distinct = 0;
  bitset<26> distinct;
  scanf("%s", &chara);
  for (int i = 0; i < strlen(chara); i++) {
    int x = ((int)chara[i]) - 97;
    if (distinct[x] == false) {
      total_distinct += 1;
      distinct[x] = true;
    }
  }
  if (total_distinct % 2 == 0)
    printf("CHAT WITH HER!\n");
  else
    printf("IGNORE HIM!\n");
}