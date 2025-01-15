#include <bits/stdc++.h>

using namespace std;

void task() {

  string a;
  string b;

  cin >> a;
  cin >> b;

  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);

  int cmp = a.compare(b);

  if (cmp < 0) {
    cout << "-1\n";
  } else if (cmp == 0) {
    cout << "0\n";
  } else {
    cout << "1\n";
  }
}

int main() {

  task();

  return 0;
}