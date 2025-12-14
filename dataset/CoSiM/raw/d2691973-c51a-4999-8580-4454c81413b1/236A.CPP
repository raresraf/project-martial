#include <bits/stdc++.h>
using namespace std;

int main() {
  int num[26], i;
  for (i = 0; i < 26; i++) {
    num[i] = 0;
  }
  char arr[101];
  cin.getline(arr, 101);
  int co = 0, x, len = strlen(arr);
  for (i = 0; i < len; i++) {
    x = arr[i] - 'a';
    num[x]++;
  }
  for (i = 0; i < 26; i++) {
    if (num[i] > 0) {
      co++;
    }
  }
  if (co % 2 == 0) {
    printf("CHAT WITH HER!\n");
  } else {
    printf("IGNORE HIM!\n");
  }
}
