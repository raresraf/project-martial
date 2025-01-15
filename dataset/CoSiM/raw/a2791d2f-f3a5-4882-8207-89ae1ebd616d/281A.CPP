#include <bits/stdc++.h>
using namespace std;
int main() {
  char arr[1001];
  scanf("%s", &arr);
  if (arr[0] >= 97 && arr[0] <= 122) {
    arr[0] = 'A' + (arr[0] - 'a');
  }
  printf("%s\n", arr);
  return 0;
}
