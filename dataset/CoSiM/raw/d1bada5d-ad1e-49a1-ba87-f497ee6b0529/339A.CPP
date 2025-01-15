#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  string word;
  vector<char> vec;

  cin >> word;

  for (auto w : word) {
    if (w == '+')
      continue;

    vec.push_back(w);
  }

  sort(vec.begin(), vec.end());

  const auto length = vec.size();

  cout << vec[0];

  for (auto i = 1; i < length; i++) {
    cout << '+' << vec[i];
  }

  cout << endl;

  return 0;
}
