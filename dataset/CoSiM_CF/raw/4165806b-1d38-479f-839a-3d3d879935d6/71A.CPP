#include <bits/stdc++.h>
using namespace std;

int main() {
  vector<string> input;
  int n;
  cin >> n;
  int lens[n];

  for (int i = 0; i < n; i++) {
    string inp;
    cin >> inp;
    input.push_back(inp);
    lens[i] = input[i].length();
  }

  for (int i = 0; i < n; i++) {
    if (lens[i] > 10) {
      cout << input[i][0] << lens[i] - 2 << input[i][input[i].length() - 1]
           << endl;
    } else {
      cout << input[i] << endl;
    }
  }
}
