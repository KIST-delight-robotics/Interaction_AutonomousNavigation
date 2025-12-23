#include "sk_robot_lib/skSubMsg.h"

template < class cMsg >
skSubMsg< cMsg >::skSubMsg() : m_msg(), m_have_msg(false), m_avg_filter(false), p_node(NULL)
{
}

template < class cMsg >
skSubMsg< cMsg >::~skSubMsg()
{
}

template < class cMsg >
void skSubMsg< cMsg >::msgCallback(const class cMsg::SharedPtr msg)
{
    this->m_msg = (*msg);
    this->m_have_msg = true;
    if( this->m_avg_filter )
        this->saveMsgFilter();
}

template < class cMsg >
void skSubMsg< cMsg >::saveMsgFilter()
{
}

template < class cMsg >
void skSubMsg< cMsg >::activateAvgFilter(const int length/* = 10*/)
{
}

template < class cMsg >
void skSubMsg< cMsg >::initialize(rclcpp::Node* node, const std::string param, const std::string name)
{
    rclcpp::Parameter p;

    this->p_node = node;

    if( this->p_node->get_parameter(param, p) )
    {
        this->initialize(p_node,p.as_string());
    }
    else
    {
        this->initialize(p_node, name);
    }
}

template < class cMsg >
void skSubMsg< cMsg >::initialize(rclcpp::Node* node, const std::string name)
{
    this->p_node = node;
    this->p_sub = this->p_node->create_subscription< cMsg >(name, 10, std::bind(&skSubMsg::msgCallback, this, _1));
}

template < class cMsg >
bool skSubMsg< cMsg >::haveMsg() const
{
    return (this->m_have_msg);
}

template < class cMsg >
void skSubMsg< cMsg >::resetMsg()
{
    this->m_have_msg = false;
}

template < class cMsg >
cMsg skSubMsg< cMsg >::getMsg() const
{
    return (this->m_msg);
}
