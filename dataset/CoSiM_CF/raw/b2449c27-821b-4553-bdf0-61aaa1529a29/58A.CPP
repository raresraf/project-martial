#include <iostream>
using namespace std;

int main() {
  string a, s;
  s = "hello";
  cin >> a;
  int l = a.size();
  int i = 0, j = 0;
  while (i < 5 && j < l) {
    if (a[j] == s[i])
      i++;
    j++;
  }
  if (i == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
