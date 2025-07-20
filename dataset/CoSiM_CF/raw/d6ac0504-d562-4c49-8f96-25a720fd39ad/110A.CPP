#include <bits/stdc++.h>
using namespace std;

int main() {
  string n;
  cin >> n;
  int len = n.length();
  int h_counter = 0;
  for (int i = 0; i < len; i++) {
    if (n[i] == '4' || n[i] == '7')
      h_counter++;
  }

  if (h_counter == 4 || h_counter == 7)
    cout << "YES";
  else
    cout << "NO";
}
