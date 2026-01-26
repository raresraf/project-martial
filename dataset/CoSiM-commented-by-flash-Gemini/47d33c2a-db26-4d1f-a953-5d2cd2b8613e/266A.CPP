#include <bits/stdc++.h>
using namespace std;
int main() {
  int n = 0, c = 0;
  cin >> n;
  char ch[n];
  for (int i = 0; i < n; i++)
    cin >> ch[i];

  for (int i = 0; i < n; i++)
    if (ch[i] == ch[i + 1])
      c++;

  cout << c;

  return 0;
}
