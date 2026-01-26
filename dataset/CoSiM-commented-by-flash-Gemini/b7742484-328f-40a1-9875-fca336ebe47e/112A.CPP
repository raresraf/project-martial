#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.size(); i++)
    a[i] = tolower(a[i]);

  for (int i = 0; i < b.size(); i++)
    b[i] = tolower(b[i]);
  if (a > b)
    cout << 1;
  else if (a == b)
    cout << 0;
  else
    cout << -1;
  return 0;
}