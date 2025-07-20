#include <iostream>
using namespace std;

int main() {
  string a;
  int i, j, n, d = 0;
  cin >> a;
  n = a.size();
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++)
      if (a[i] == a[j])
        break;
    if (j == n)
      d++;
  }
  if (d % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
