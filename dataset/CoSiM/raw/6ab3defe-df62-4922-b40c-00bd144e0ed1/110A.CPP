#include <bits/stdc++.h>

using namespace std;

void analyze(string s);

int main() {
  string number;

  while (cin >> number) {
    analyze(number);
  }

  return 0;
}

void analyze(string s) {
  int counter = 0;

  for (int i = 0; i < s.size(); ++i) {
    if ((s[i] == '4') || (s[i] == '7'))
      ++counter;
  }

  if ((counter == 4) || (counter == 7))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}