#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  char c;
  vector<int> sum;

  while (cin.get(c)) {
    if (c - 48 > 0)
      sum.push_back(c - 48);
  }

  sort(sum.begin(), sum.end(), [](int a, int b) { return a < b; });
  vector<int>::iterator it = sum.begin();
  cout << *it;
  it++;
  while (it != sum.end()) {
    cout << "+" << *it;
    it++;
  }

  return 0;
}