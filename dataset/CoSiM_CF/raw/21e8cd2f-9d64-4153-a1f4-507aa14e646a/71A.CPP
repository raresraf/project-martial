#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  cin >> n;
  string m[200];
  for (int i = 0; i < n; i++) {
    cin >> m[i];
  }
  for (int i = 0; i < n; i++) {
    if (m[i].size() > 10)
      cout << *m[i].begin() << m[i].length() - 2 << *m[i].rbegin() << endl;
    else
      cout << m[i] << endl;
  }

  return 0;
}
