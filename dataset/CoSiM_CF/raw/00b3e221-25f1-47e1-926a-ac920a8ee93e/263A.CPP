#include <iostream>
#include <string>
#include <vector>

using namespace std;

bool doesOneExist(string c) {
  if (c.find('1') != std::string::npos) {
    return true;
  }
  return false;
}

int main() {
  string a;
  string result;
  int n = 5, y = 0, cy = 0;
  while (n--) {
    getline(cin, a);
    if (doesOneExist(a)) {
      cy = y;
      result = a;
    }
    y++;
  }

  y = cy;
  int moves = abs(2 - y);

  vector<int> x;
  for (char &c : result) {
    if (c == '0') {
      x.push_back(0);
    } else if (c == '1') {
      x.push_back(1);
    }
  }

  for (int i = 0; i < x.size(); i++) {
    if (x[i] == 1)
      moves += abs(2 - i);
  }
  cout << moves << endl;
}