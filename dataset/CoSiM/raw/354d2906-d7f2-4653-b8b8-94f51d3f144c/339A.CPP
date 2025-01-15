#include <bits/stdc++.h>
using namespace std;

int main() {
  char s[100];
  cin >> s;
  int j = 0;
  int n = strlen(s);
  int num[50];
  for (int i = 0; i < n; i += 2) {
    int k = s[i];
    k -= 48;
    num[j++] = k;
  }
  sort(num, num + j);
  for (int i = 0; i < j; i++) {
    cout << num[i];
    if (i != j - 1)
      cout << "+";
  }
  cout << endl;
  return 0;
}
