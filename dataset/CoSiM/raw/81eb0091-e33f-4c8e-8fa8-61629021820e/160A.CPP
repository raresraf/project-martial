#include <iostream>
#include <queue>

using std::cin;
using std::cout;
using std::endl;

int main() {
  int n, coinsNumber = 0;
  int twinMoney = 0, myMoney = 0;
  std::priority_queue<int> q;
  cin >> n;
  for (int val, i = 0; i != n; ++i) {
    cin >> val;
    twinMoney += val;
    q.push(val);
  }

  while (myMoney <= twinMoney) {
    myMoney += q.top();
    twinMoney -= q.top();
    q.pop();
    coinsNumber++;
  }

  cout << coinsNumber;
  return 0;
}
