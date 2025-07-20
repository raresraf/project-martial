/*#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
using namespace std;

int main(){
    char str[100]={0};
    int len=0,f=0;
        fgets(str,100,stdin);
        len=strlen(str);
        if(str[len-1]!='\0'){
            str[len-1]='\0';
            len--;
        }
        for(int i=0;i<len;i++){
            if(str[i]==str[i+1]){
                f++;
                if(f==6){
                    cout<<"YES"<<endl;
                    break;
                }
            }
            else f=1;
        }
        if(f<6) cout<<"NO"<<endl;
    return 0;
}

#include<cstdio>
#include<cstring>
#include<iostream>
#include<string>
using namespace std;

int main(){
    char str[100]={0};
    int len=0,f=0,c1=0,c2=0;
        fgets(str,100,stdin);
        len=strlen(str);
        if(str[len-1]!='\0'){
            str[len-1]='\0';
            len--;
        }
        for(int i=0;i<len;i++){
            if(str[i]=='0')
             c1++;


            else c2++;
        }
        if(c1==7 || c2==7)
         { f=1;
         }
if(f==1){
cout<<"yes"<<endl;}
else{
cout<<"no"<<endl;}
    return 0;
}

#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
    int flag, f;
    char s[105];
    scanf("%s", s);
    f=flag=0;
    for(int i=0; i<strlen(s); i++) {
        if(s[i]==s[i+1])
            f++;
        else f=0;
        if(f==6) {
            flag=1;
            printf("YES");
            break;
        }
    }
    if(flag==0) printf("NO");
    return 0;
}*/

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;

int main() {
  char str[100] = {0};
  int len = 0, f = 0, flag = 0;
  fgets(str, 100, stdin);
  len = strlen(str);
  if (str[len - 1] != '\0') {
    str[len - 1] = '\0';
    len--;
  }
  for (int i = 0; i < len; i++) {
    if (str[i] == str[i + 1])
      f++;
    else
      f = 0;

    if (f == 6) {
      flag = 1;
      cout << "YES" << endl;
      break;
    }
  }

  if (flag == 0) {
    cout << "No" << endl;
  }

  return 0;
}
