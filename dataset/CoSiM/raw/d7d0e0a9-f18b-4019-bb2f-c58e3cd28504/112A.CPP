/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>

using namespace std;

int main() {
  int t = 0;
  string a;
  cin >> a;
  char c;

  while (a[t]) {
    c = a[t];
    a[t] = (tolower(c));
    t++;
  }
  // int j =0;
  string b;
  cin >> b;
  t = 0;
  while (b[t]) {
    c = b[t];
    b[t] = (tolower(c));
    t++;
  }
  for (int u = 0; u < a.length(); u++) {
    if (a[u] > b[u]) {
      cout << 1 << endl;
      break;
    } else if (a[u] < b[u]) {
      cout << -1 << endl;
      break;
    } else if ((a[u] == b[u]) && (u == (a.length() - 1)))
      cout << 0;
  }
  return 0;
}
