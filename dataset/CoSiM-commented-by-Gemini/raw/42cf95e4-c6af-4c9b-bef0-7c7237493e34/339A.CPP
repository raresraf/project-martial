#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int conver(char a) { return a - 48; }

int main() {
  string a;
  vector<int> num_char2;
  cin >> a;
  for (int i = 0; i < a.length(); i = i + 2) {
    num_char2.push_back(conver(a[i]));
  }
  sort(num_char2.begin(), num_char2.end());
  for (vector<int>::iterator i = num_char2.begin(); i != num_char2.end(); i++) {
    if (i == num_char2.end() - 1) {
      cout << *i;
    } else {
      cout << *i << "+";
    }
  }
}