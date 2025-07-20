#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, co = 0;
  scanf("%d", &n);
  getchar();
  char arr[51];
  cin.getline(arr, 51);
  int len = strlen(arr);
  for (i = 0; i < len - 1; i++) {
    if (arr[i] == arr[i + 1]) {
      co++;
    }
  }
  printf("%d\n", co);
}
