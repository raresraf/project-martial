#include <algorithm>
#include <bits/stdc++.h>
#include <vector>

using namespace std;

int main() {
  int i;
  string input;
  while (getline(cin, input)) {
    vector<char> num;
    for (i = 0; i < input.length(); i = i + 2) {
      num.push_back(input[i]);
    }
    sort(num.begin(), num.begin() + num.size());
    for (i = 0; i < num.size() - 1; i++) {
      cout << num[i] << "+";
    }
    cout << num[i] << endl;
    return 0;
  }
  return 0;
}
