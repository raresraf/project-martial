#include <bits/stdc++.h>
using namespace std;

int main() {
  long long num;
  cin >> num;

  string str = to_string(num);
  int count = 0;
  for (char c : str) {
    int l = (int)c - 48;

    if (l == 4 || l == 7)
      count++;
  }

  regex pattern("[47]*");
  string nl = to_string(count);

  regex_match(nl, pattern) ? cout << "YES" : cout << "NO";
  cout << endl;

  return 0;
}
