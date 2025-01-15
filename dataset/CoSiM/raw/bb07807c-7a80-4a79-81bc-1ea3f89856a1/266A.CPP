#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int count = 0;

  char s[100];

  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  for (int i = 0; i < strlen(s); i++) {
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  cout << count << endl;
}
