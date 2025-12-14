#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  string x;
  cin >> x;
  int siz = x.size();
  vector<char> vec;
  for (int j = 0; j < siz;) {
    vec.push_back(x[j]);
    j += 2;
  }
  sort(vec.begin(), vec.end());
  siz = vec.size();
  for (int j = 0; j < siz; j++) {
    cout << vec[j];
    if (j + 1 < siz) {
      cout << "+";
    }
  }
  return 0;
}
