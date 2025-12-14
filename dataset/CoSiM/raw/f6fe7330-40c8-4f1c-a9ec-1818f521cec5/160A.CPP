#include <algorithm> // sort
#include <iostream>
#include <stdlib.h> //exit
#include <vector>

using namespace std;

int main() {
  int a, b;
  int sum = 0, sum2 = 0, cnt = 0;
  vector<int> ele;
  cin >> a;
  while (a--) {
    cin >> b;
    ele.insert(ele.end(), b);
  }
  for (vector<int>::iterator it = ele.begin(); it != ele.end(); ++it)
    sum += *it;
  sort(ele.begin(), ele.end());
  // for(vector<int>::iterator it=ele.begin(); it!=ele.end();++it) {cout<<*it;}

  for (vector<int>::iterator i = (ele.end() - 1); i >= ele.begin(); i--) {
    // cout<<*i;
    sum2 += (*i);
    cnt++;
    if (sum2 > (sum / 2)) {
      break;
    }
  }
  cout << cnt;
}
