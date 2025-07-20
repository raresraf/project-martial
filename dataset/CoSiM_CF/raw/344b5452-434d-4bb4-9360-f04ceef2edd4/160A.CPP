#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;
int main() {
  int n;
  cin >> n;
  vector<int> vec;
  while (n--) {
    int a;
    cin >> a;
    vec.push_back(a);
  }
  sort(vec.rbegin(), vec.rend());

  auto total = accumulate(vec.begin(), vec.end(), 0);
  auto current = 0;
  auto count = 0;
  for (auto i = 0; i != vec.size(); i++) {
    current += vec[i];
    total -= vec[i];
    count++;
    if (current > total)
      break;
  }
  cout << count;
}