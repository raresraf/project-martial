#include <bits/stdc++.h>
using namespace std;

#define pb push_back
#define vi vector<int>

/*
bool divideeven(int w) {
        //logic:
        //get all the even numbers from 1 to w in a avector
        //and then subtract taht tfrom w and check if %2==0
        bool status = false;

        vi even_list;
        for (int i = 1; i <= w; i++) {
                if (i % 2 == 0) {
                        even_list.pb(i);
                }
        }

        for (size_t i = 0; i < even_list.size(); i++) {
                if ((w - even_list[i]) % 2 == 0 ) {
                        status = true;
                        break;
                }
        }
        return status;

}

*/
int main() {
  int w;
  cin >> w;
  /*if (divideeven(w)) {
          cout << "YES" << "\n";
  }
  else {
          cout << "NO" << "\n";
  }
  */
  if (w % 2 == 0 && w > 2) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
