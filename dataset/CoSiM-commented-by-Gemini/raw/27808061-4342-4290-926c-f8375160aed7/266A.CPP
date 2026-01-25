#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;

  string str;
  cin >> str;
  int count = 0;

  for (int i = 1; i < str.size(); i++) {
    if (str[i] == str[i - 1]) {
      str.erase(str.begin() + i - 1);
      i--;
      count++;
    }
  }

  cout << count;
  return 0;
}
