#include <bits/stdc++.h>

using namespace std;

int main() {

  string t;
  cin >> t;

  cout << ((t.find("0000000") != string::npos ||
            t.find("1111111") != string::npos)
               ? "YES"
               : "NO");
}