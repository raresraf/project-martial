#include <algorithm>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  char a[100];
  cin >> a;
  int b = strlen(a);
  sort(a, a + b);
  int i, j = 0;
  for (i = 0; i < b; i++) {
    if (a[i] != a[i + 1]) {
      j++;
    }
  }

  if (j % 2 == 0) {
    cout << "CHAT WITH HER!";
  }
  if (j % 2 != 0) {
    cout << "IGNORE HIM!";
  }
}
