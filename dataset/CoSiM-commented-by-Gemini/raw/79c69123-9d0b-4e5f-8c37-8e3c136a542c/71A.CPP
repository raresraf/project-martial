#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  vector<string> use_case;

  for (int i = 0; i < t; i++) {
    string h, c1, c3;
    cin >> h;
    c1 = h[0];
    int j = h.length();
    std::string c2 = std::to_string(j);
    c3 = h[j - 1];
    c2.append(c3);
    c1.append(c2);
    if (h.length() >= 10) {
      use_case.push_back(c1);
    }
  }
  for (int i = 0; i < use_case.size(); i++) {
    cout << use_case[i] << endl;
  }
  return 0;
}
