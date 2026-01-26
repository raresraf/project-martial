#include <iostream>
#include <string>
using namespace std;
int main() {
  string a;
  int c;
  cin >> a;
  c = 1;
  for (int i = 0, j = 1; j < a.length(); j++, i++) {

    if (a[i] == a[j])
      ++c;
    else
      c = 1;
    if (c == 7)
      break;
  }
  if (c == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
