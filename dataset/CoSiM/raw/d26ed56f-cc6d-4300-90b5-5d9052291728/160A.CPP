#include <bits/stdc++.h>
using namespace std;
int sum(const vector<int> &, int, int);

int main(void) {
  int n;
  cin >> n;
  vector<int> coins;

  while (n--) {
    int input;
    cin >> input;
    coins.push_back(input);
  }

  sort(coins.begin(), coins.end());
  int last_pos, sum_of_after, sum_of_before, i, j, count;

  last_pos = coins.size() - 1;
  sum_of_after = sum(coins, last_pos, last_pos);
  sum_of_before = sum(coins, 0, last_pos - 1);

  i = last_pos, j = last_pos - 1, count = 1;

  while (sum_of_before >= sum_of_after) {
    count++;
    sum_of_after = sum(coins, --i, last_pos);
    sum_of_before = sum(coins, 0, --j);
  }

  cout << count << endl;
  return 0;
}

int sum(const vector<int> &data, int start, int end) {
  int result = 0;
  for (int i = start; i <= end; i++)
    result += data.at(i);
  return result;
}
